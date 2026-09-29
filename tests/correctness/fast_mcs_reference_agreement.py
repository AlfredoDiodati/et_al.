"""Checks inference/fast_mcs.h against the authors' own implementation.

    make test-fast-mcs-python FASTMCS_REFERENCE=path/to/fastMCS     (PYTHON=... for an interpreter with numpy)

Outside make test, since it needs Python with numpy and a clone of
https://github.com/Sylvain-Barde/fastMCS, the implementation Barde (2026)
links. FASTMCS_REFERENCE is that clone's directory, or its fastMCS.py; the
commit it is at is printed.

What is compared. Both sides get the same losses and the same resamples:
the row indices mcs() and fast_mcs() draw are taken from this library
(c_draws) and handed to the reference in place of its own block bootstrap,
whose blocks wrap around the end of the sample and are drawn from numpy,
so no seed could make the two schemes produce the same draws. Everything
downstream of the draws is each side's own code. Then:

1. Control: the reference's elimination algorithm (algorithm='elimination')
   against mcs(). Both compute the Hansen, Lunde and Nason procedure
   exactly, so they must agree: the same elimination order, each round's
   statistic to 1e-9 relative, the same number of draws exceeding it. If
   this fails, the two sides are not seeing the same inputs and nothing
   below means anything.
2. The reference's one-pass algorithm (algorithm='1-pass') as published,
   against fast_mcs(): how often the order, the counts and the set differ.
   Reported, not asserted.
3. The same with one line of the reference changed: where it adds a model
   ranked below the new one to the running maximum of |tau|, it takes the
   models in the order they were added and assigns the result by ranking
   position; the change takes them in ranking order, which is Eq. (18) of
   the paper. That version must agree with fast_mcs() exactly as the
   control does. The reference's other three departures from the printed
   equations (docs/FAST_MCS_DOCUMENTATION.md) are left in: in the order
   the one-pass algorithm adds models they cannot act, and this checks
   that too.
4. The reference's two-pass algorithm against mcs(), reported.
5. Which one-pass is the error. In the order the one-pass algorithm adds
   models, its heuristic (Eqs. (28)-(29)) never runs and every update of
   a model's bootstrapped statistic is Eq. (18), which is exact, so a
   correct one-pass returns what the exact algorithms return. For each
   replication where the published one-pass and fast_mcs() disagree, this
   checks which of them agrees with the reference's own exact two-pass
   algorithm and with mcs(). Reported, and also asserted for fast_mcs():
   wherever the two disagree, fast_mcs() must be the one that equals both
   exact algorithms.

Counts are compared rather than p-values: a p-value here is a count of
draws over the draw count, the reference counts ties as exceeding (>=)
and this library does not (>), so the reference's counts are recomputed
with > from its own bootstrapped statistics (tBootDist) and statistics
(tScore). A reported p-value difference is then an algorithmic one.

Designs: the design of Hansen, Lunde and Nason (2011) that Barde (2026)
uses, Eqs. (30)-(31), from numpy's default_rng(2026): per replication
lambda uniform on [5, 40], rho on [0, 0.95], phi on [0, 0.8], columns
shuffled; T of 250 and 30, M of 10, 50 and 100, 500 draws, blocks of two,
alpha 0.05. FASTMCS_REPLICATIONS sets the replications per (T, M), 20 by
default.
"""
import ctypes
import importlib.util
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libfastmcs_f64.so"], cwd=ROOT, check=True)

reference_path = os.environ.get("FASTMCS_REFERENCE")
if not reference_path:
    sys.exit("FASTMCS_REFERENCE is not set: clone https://github.com/Sylvain-Barde/fastMCS and pass its directory")
if os.path.isdir(reference_path):
    reference_path = os.path.join(reference_path, "fastMCS.py")
if not os.path.isfile(reference_path):
    sys.exit("no fastMCS.py at " + reference_path)
reference_dir = os.path.dirname(os.path.abspath(reference_path))
commit = subprocess.run(["git", "-C", reference_dir, "log", "-1", "--format=%H %cd"],
                        capture_output=True, text=True).stdout.strip() or "not a git checkout"

REPLICATIONS = int(os.environ.get("FASTMCS_REPLICATIONS", "20"))
B, BLOCK, ALPHA = 500, 2, 0.05

source = open(reference_path).read()
PUBLISHED_LINE = "tMax = np.maximum(tMax, np.abs(tBoot[:,mod]))"
RANKED_LINE = ("tMax = np.maximum(tMax, np.abs(tBoot[:,int(np.where("
               "modsProcessed[:-1] == modRank[j])[0][0])]))")
if source.count(PUBLISHED_LINE) != 1:
    sys.exit("the reference no longer contains the line this test changes, exactly once: " + PUBLISHED_LINE)


def load(name, text):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(text, reference_path, "exec"), module.__dict__)
    return module


published = load("fastMCS_published", source)
ranked = load("fastMCS_ranked_running_max", source.replace(PUBLISHED_LINE, RANKED_LINE))

lib = ctypes.CDLL(os.path.join(ROOT, "libfastmcs_f64.so"))
assert lib.c_is_double() == 1
I, U, D = ctypes.c_int, ctypes.c_ulonglong, ctypes.c_double
PD, PI = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
lib.c_draws.argtypes = [I, I, I, U, U, PI]
lib.c_draws.restype = None
lib.c_run.argtypes = [I, I, I, PD, I, I, U, U, D, PI, PD, PD, PI]
lib.c_run.restype = I


def ours(which, L, seed):
    n, M = L.shape
    Lc = np.ascontiguousarray(L, np.float64)
    order = np.empty(M - 1, np.int32)
    statistic = np.empty(M - 1)
    pvalue = np.empty(M - 1)
    surviving = np.empty(M, np.int32)
    k = lib.c_run(which, n, M, Lc.ctypes.data_as(PD), B, BLOCK, seed, 0, ALPHA, order.ctypes.data_as(PI),
                  statistic.ctypes.data_as(PD), pvalue.ctypes.data_as(PD), surviving.ctypes.data_as(PI))
    return order, statistic, np.rint(pvalue * B).astype(int), np.sort(surviving[:k])


def draws(n, seed):
    out = np.empty(B * n, np.int32)
    lib.c_draws(n, B, BLOCK, seed, 0, out.ctypes.data_as(PI))
    return out.reshape(B, n).T.copy()    # the reference holds them obs x B


def theirs(module, algorithm, L, indices):
    """The reference on L, resampling with this library's indices."""
    module.blockBootstrap = lambda rng, obs, draws_wanted, b: indices
    m = module.mcs(seed=0, verbose=False)
    m.addLosses(np.array(L, np.float64))
    m.run(B=B, b=BLOCK, bootstrap="block", algorithm=algorithm)
    included, _ = m.getMCS(alpha=ALPHA)
    M = L.shape[1]
    order = np.asarray(m.exclMods[:M - 1], int)
    statistic = m.tScore[order]
    exceed = np.array([np.sum(m.tBootDist[:, k] > m.tScore[k]) for k in order])
    return order, statistic, exceed, np.sort(np.asarray(included, int))


def hln(rng, N, M):
    """Barde (2026) Eqs. (30)-(31)."""
    lam, rho, phi = 5 + 35 * rng.random(), 0.95 * rng.random(), 0.8 * rng.random()
    theta = lam / np.sqrt(N) * np.concatenate(([0.0], np.arange(M - 2) / (M - 1), [1.0]))
    y, c = 0.0, -phi / (2 * (1 + phi))
    a = np.empty(N)
    for n in range(-50, N):
        y = c + phi * y + np.sqrt(phi) * rng.standard_normal()
        if n >= 0:
            a[n] = np.exp(y) / np.sqrt(np.exp(phi / (1 - phi * phi)))
    X = np.sqrt(rho) * rng.standard_normal((N, 1)) + np.sqrt(1 - rho) * rng.standard_normal((N, M))
    L = theta[None, :] + a[:, None] * X
    return L[:, rng.permutation(M)]


def agree(a, b):
    """Discrete agreement of two (order, statistic, exceed, set) tuples, and
    the largest relative gap between their statistics."""
    same_order = np.array_equal(a[0], b[0])
    gap = np.max(np.abs(a[1] - b[1]) / np.maximum(1.0, np.abs(b[1]))) if same_order else np.inf
    return same_order, np.array_equal(a[2], b[2]), np.array_equal(a[3], b[3]), gap


print("inference/fast_mcs.h against " + reference_path)
print("  reference commit " + commit)
print("  numpy " + np.__version__ + ", %d replications per design, %d draws, blocks of %d, alpha %.2f"
      % (REPLICATIONS, B, BLOCK, ALPHA))
print()
print("  %4s %4s | %-26s | %-34s | %-26s | %-24s" % ("T", "M", "control: elimination", "one-pass as published",
                                                      "one-pass, ranked maximum", "two-pass vs mcs"))
failures = 0
exactness = []
rng = np.random.default_rng(2026)
for N in (250, 30):
    for M in (10, 50, 100):
        counts = {"control": 0, "published_order": 0, "published_counts": 0, "published_set": 0,
                  "ranked": 0, "two_pass": 0, "disagree": 0, "published_exact": 0, "ours_exact": 0,
                  "published_vs_own": 0, "ours_vs_own": 0}
        largest_gap = 0.0
        for r in range(REPLICATIONS):
            L = hln(rng, N, M)
            seed = 1000 * N + 10 * M + r
            indices = draws(N, seed)
            elimination = ours(0, L, seed)
            fast = ours(1, L, seed)

            o, e, s, gap = agree(theirs(published, "elimination", L, indices), elimination)
            ok = o and e and s and gap <= 1e-9
            if ok:
                counts["control"] += 1
                largest_gap = max(largest_gap, gap)
            else:
                failures += 1
                print("  FAIL control, T %d M %d replication %d: order %s, counts %s, set %s, gap %g"
                      % (N, M, r, o, e, s, gap))

            published_one_pass = theirs(published, "1-pass", L, indices)
            o, e, s, gap = agree(published_one_pass, fast)
            counts["published_order"] += not o
            counts["published_counts"] += not e
            counts["published_set"] += not s
            disagree = not (o and e and s)

            o, e, s, gap = agree(theirs(ranked, "1-pass", L, indices), fast)
            if o and e and s and gap <= 1e-9:
                counts["ranked"] += 1
                largest_gap = max(largest_gap, gap)
            else:
                failures += 1
                print("  FAIL ranked one-pass, T %d M %d replication %d: order %s, counts %s, set %s, gap %g"
                      % (N, M, r, o, e, s, gap))

            their_two_pass = theirs(published, "2-pass", L, indices)
            o, e, s, gap = agree(their_two_pass, elimination)
            counts["two_pass"] += not (o and e and s)

            published_matches_own = all(agree(published_one_pass, their_two_pass)[:3])
            ours_matches_own = all(agree(fast, elimination)[:3])
            counts["published_vs_own"] += not published_matches_own
            counts["ours_vs_own"] += not ours_matches_own
            if disagree:
                counts["disagree"] += 1
                counts["published_exact"] += published_matches_own
                ours_exact = ours_matches_own and all(agree(fast, their_two_pass)[:3])
                counts["ours_exact"] += ours_exact
                if not ours_exact:
                    failures += 1
                    print("  FAIL T %d M %d replication %d: fast_mcs and the published one-pass disagree and"
                          " fast_mcs does not equal the exact algorithms" % (N, M, r))

        print("  %4d %4d | %3d of %3d agree            | order %3d, counts %3d, set %3d differ | "
              "%3d of %3d agree          | %3d of %3d differ"
              % (N, M, counts["control"], REPLICATIONS, counts["published_order"], counts["published_counts"],
                 counts["published_set"], counts["ranked"], REPLICATIONS, counts["two_pass"], REPLICATIONS))
        exactness.append((N, M, counts))
        sys.stdout.flush()

print()
print("  which one-pass equals the exact algorithms (the reference's two-pass, and mcs())")
print("  %4s %4s | %-30s | %-30s | %-22s | %-22s" % ("T", "M", "published one-pass vs its own", "fast_mcs vs mcs",
                                                    "the two one-pass differ", "of those, exact is"))
for N, M, c in exactness:
    print("  %4d %4d | %3d of %3d differ              | %3d of %3d differ              | %3d of %3d             |"
          " fast_mcs %3d, published %3d"
          % (N, M, c["published_vs_own"], REPLICATIONS, c["ours_vs_own"], REPLICATIONS, c["disagree"],
             REPLICATIONS, c["ours_exact"], c["published_exact"]))
print()
print("  largest relative gap between agreeing statistics: %.3g" % largest_gap)
print("  control and ranked one-pass must agree in every replication; the published one-pass")
print("  and the two-pass columns count replications that differ in elimination order, in the")
print("  number of draws exceeding some round's statistic, or in the set at alpha %.2f" % ALPHA)
print()
print("%s, %d failures" % ("FAILED" if failures else "PASSED", failures))
sys.exit(1 if failures else 0)
